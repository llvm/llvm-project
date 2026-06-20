/* ============================================================
   LLVM Advisor — Compare View
   ============================================================ */

const CompareView = {
  _baseId: null,
  _candidateId: null,
  _results: null,
  _baseSummary: null,
  _candidateSummary: null,

  async render(params) {
    this._baseId = params.base || null;
    this._candidateId = params.candidate || null;

    const container = h('div', {});
    container.appendChild(h('div', { class: 'compare-header' },
      this.snapshotSelect('BASE', this._baseId, id => { this._baseId = id; this.runCompare(); }),
      h('span', { class: 'arrow mono' }, '→'),
      this.snapshotSelect('CANDIDATE', this._candidateId, id => { this._candidateId = id; this.runCompare(); }),
    ));

    container.appendChild(h('div', { id: 'compare-summary-section' }));
    container.appendChild(h('div', { class: 'summary-bar', id: 'compare-summary' }));
    container.appendChild(h('div', { class: 'section-header' }, 'Unit Changes'));
    container.appendChild(h('div', { id: 'compare-results' }));

    Shell.renderMain(container);
    if (this._baseId && this._candidateId) this.runCompare();
  },

  snapshotSelect(label, selectedId, onChange) {
    const snaps = State.get('snapshots') || [];
    const dd = h('div', { class: 'dropdown' });
    const display = selectedId ? (selectedId.slice(0, 8) + '…') : `Select ${label}`;
    const trigger = h('button', { class: 'dd-trigger', onClick: e => Shell.toggleDropdown(e) },
      `${label}: ${display} ▾`);
    const menu = h('div', { class: 'dd-menu' });
    snaps.forEach(s => {
      menu.appendChild(h('div', {
        class: 'dd-item' + (s.id === selectedId ? ' selected' : ''),
        onClick: () => { onChange(s.id); Shell.closeDropdowns(); trigger.textContent = `${label}: ${s.id.slice(0, 8)}… ▾`; }
      }, `${s.id.slice(0, 8)} · ${timeAgo(s.created_unix)}`));
    });
    dd.appendChild(trigger);
    dd.appendChild(menu);
    return dd;
  },

  async runCompare() {
    if (!this._baseId || !this._candidateId) return;
    window.location.hash = `#/compare?base=${encodeURIComponent(this._baseId)}&candidate=${encodeURIComponent(this._candidateId)}`;

    const resultsEl = document.getElementById('compare-results');
    const summaryEl = document.getElementById('compare-summary');
    const sumSectionEl = document.getElementById('compare-summary-section');
    if (resultsEl) { clearEl(resultsEl); resultsEl.appendChild(h('div', { class: 'text-muted mono', style: { padding: '16px' } }, 'Comparing…')); }

    const [compareRes, baseSumRes, candSumRes] = await Promise.all([
      API.compare(this._baseId, this._candidateId),
      API.snapshotSummary(this._baseId).catch(() => ({ ok: false, data: {} })),
      API.snapshotSummary(this._candidateId).catch(() => ({ ok: false, data: {} })),
    ]);

    this._baseSummary = baseSumRes.ok && baseSumRes.data ? baseSumRes.data : {};
    this._candidateSummary = candSumRes.ok && candSumRes.data ? candSumRes.data : {};

    if (!compareRes.ok) {
      if (resultsEl) { clearEl(resultsEl); resultsEl.appendChild(UI.errorCard(compareRes.error || 'Compare failed', () => this.runCompare())); }
      return;
    }

    this._results = compareRes.data;
    this.renderMatchSummary(summaryEl);
    this.renderSummaryComparison(sumSectionEl);
    this.renderResults(resultsEl);
  },

  renderMatchSummary(el) {
    if (!el || !this._results) return;
    clearEl(el);
    const summary = this._results.match_summary || {};
    [
      { label: 'Matched', value: summary.matched ?? 0, tone: 'neutral' },
      { label: 'Changed', value: summary.changed ?? 0, tone: 'warn' },
      { label: 'Added', value: summary.added ?? 0, tone: 'info' },
      { label: 'Removed', value: summary.removed ?? 0, tone: 'danger' },
    ].forEach(m => {
      const card = h('div', { class: `summary-metric${m.value > 0 && m.tone !== 'neutral' ? ' ' + m.tone : ''}` },
        h('div', { class: 'label' }, m.label),
        h('div', { class: 'values' }, String(m.value))
      );
      el.appendChild(card);
    });
  },

  renderSummaryComparison(el) {
    if (!el) return;
    clearEl(el);
    const base = this._baseSummary;
    const cand = this._candidateSummary;
    if (!base.health_score && !cand.health_score) return;

    const section = h('div', { class: 'compare-summary-section' });

    // Health score before -> after
    const bHealth = base.health_score ?? 0;
    const cHealth = cand.health_score ?? 0;
    const healthDelta = cHealth - bHealth;
    const arrowCls = healthDelta > 0 ? 'up' : healthDelta < 0 ? 'down' : 'flat';
    const arrowChar = healthDelta > 0 ? '↑' : healthDelta < 0 ? '↓' : '→';

    section.appendChild(h('h3', {}, 'Snapshot Comparison'));
    section.appendChild(h('div', { class: 'compare-health-delta' },
      h('div', {},
        h('div', { style: { fontSize: '11px', color: 'var(--fg3)', marginBottom: '4px' } }, 'Health Score'),
        h('div', { style: { display: 'flex', alignItems: 'center', gap: '12px' } },
          h('span', { class: 'compare-health-value', style: { color: 'var(--fg3)' } }, String(Math.round(bHealth))),
          h('span', { class: `compare-health-arrow ${arrowCls}` }, arrowChar),
          h('span', { class: 'compare-health-value' }, String(Math.round(cHealth))),
          healthDelta !== 0 ? h('span', {
            class: `snap-delta ${healthDelta > 0 ? 'positive' : 'negative'}`,
            style: { fontSize: '12px', marginLeft: '8px' }
          }, `${healthDelta > 0 ? '+' : ''}${healthDelta.toFixed(0)}`) : null
        )
      )
    ));

    // Key metric deltas
    const metricKeys = [
      { key: 'instructions', label: 'Instructions' },
      { key: 'functions', label: 'Functions' },
      { key: 'remarks', label: 'Remarks' },
      { key: 'warnings', label: 'Warnings' },
      { key: 'errors', label: 'Errors' },
      { key: 'unit_count', label: 'Units' },
    ];

    const deltaItems = metricKeys.map(m => ({
      label: m.label,
      before: Number(base[m.key] || (base.metrics || {})[m.key] || 0),
      after: Number(cand[m.key] || (cand.metrics || {})[m.key] || 0),
    })).filter(it => it.before !== 0 || it.after !== 0);

    if (deltaItems.length) {
      section.appendChild(h('h3', { style: { marginTop: '16px' } }, 'Metric Changes'));
      section.appendChild(UI.deltaBar(deltaItems));
    }

    // Family coverage comparison
    const baseFamilies = base.families || [];
    const candFamilies = cand.families || [];
    if (baseFamilies.length || candFamilies.length) {
      section.appendChild(h('h3', { style: { marginTop: '16px' } }, 'Capability Coverage'));
      const allFams = [...new Set([...baseFamilies.map(f => f.family), ...candFamilies.map(f => f.family)])].sort();
      const famGrid = h('div', { style: { display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(150px,1fr))', gap: '10px', marginTop: '10px' } });
      allFams.forEach(fam => {
        const bf = baseFamilies.find(f => f.family === fam) || { available: 0, missing: 0 };
        const cf = candFamilies.find(f => f.family === fam) || { available: 0, missing: 0 };
        const bTotal = bf.available + (bf.missing || 0);
        const cTotal = cf.available + (cf.missing || 0);
        const bPct = bTotal > 0 ? Math.round(bf.available / bTotal * 100) : 0;
        const cPct = cTotal > 0 ? Math.round(cf.available / cTotal * 100) : 0;
        const delta = cPct - bPct;
        famGrid.appendChild(h('div', { style: { padding: '10px', background: 'var(--bg2)', borderRadius: 'var(--r)', border: '1px solid var(--border)' } },
          h('div', { style: { fontWeight: '500', fontSize: '12px', marginBottom: '6px' } }, fam),
          h('div', { style: { display: 'flex', alignItems: 'center', gap: '8px', fontFamily: 'var(--mono)', fontSize: '12px' } },
            h('span', { style: { color: 'var(--fg3)' } }, `${bPct}%`),
            h('span', { style: { color: 'var(--fg3)' } }, '→'),
            h('span', {}, `${cPct}%`),
            delta !== 0 ? h('span', { class: `snap-delta ${delta > 0 ? 'positive' : 'negative'}` }, `${delta > 0 ? '+' : ''}${delta}%`) : null
          )
        ));
      });
      section.appendChild(famGrid);
    }

    el.appendChild(section);
  },

  renderResults(el) {
    if (!el || !this._results) return;
    clearEl(el);
    const changes = Array.isArray(this._results.unit_changes) ? this._results.unit_changes : [];
    if (!changes.length) {
      el.appendChild(h('div', { class: 'empty-state' },
        h('div', {}, 'No unit-level changes detected'),
        h('div', { class: 'reason' }, 'The two snapshots have the same compilation units')
      ));
      return;
    }

    const frag = document.createDocumentFragment();
    changes.slice(0, 50).forEach((change, idx) => {
      const matchType = change.match_type || 'changed';
      const toneCls = matchType === 'added' ? 'info' : matchType === 'removed' ? 'danger' : matchType === 'changed' ? 'warn' : 'neutral';

      const diffs = (Array.isArray(change.capability_diffs) ? change.capability_diffs : [])
        .filter(d => !isCorruptedString(d.capability));
      const summaryText = diffs.length
        ? `${diffs.length} capability change${diffs.length === 1 ? '' : 's'}`
        : titleCase(matchType);

      const unitName = change.unit_name
        || (change.candidate_unit_id || change.base_unit_id || '').slice(0, 12);

      const row = h('div', {
        class: 'regression-row', id: `compare-row-${idx}`,
        onClick: () => this.toggleRowDetail(idx, change)
      },
        h('span', { class: `severity-badge ${toneCls}` }, matchType),
        h('span', { class: 'mono', style: { overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' } }, unitName),
        h('span', { class: 'text-muted' }, summaryText),
        h('span', { class: 'text-muted', style: { textAlign: 'center' } }, '›')
      );

      const detail = h('div', {
        class: 'regression-detail', id: `compare-detail-${idx}`,
        style: { display: 'none' }
      });

      frag.appendChild(row);
      frag.appendChild(detail);
    });
    el.appendChild(frag);
  },

  toggleRowDetail(idx, change) {
    const row = document.getElementById(`compare-row-${idx}`);
    const detail = document.getElementById(`compare-detail-${idx}`);
    if (!row || !detail) return;

    const isOpen = row.classList.contains('expanded');
    document.querySelectorAll('.regression-row.expanded').forEach(r => {
      if (r !== row) {
        r.classList.remove('expanded');
        const d = document.getElementById(r.id.replace('row', 'detail'));
        if (d) d.style.display = 'none';
      }
    });

    if (isOpen) {
      row.classList.remove('expanded');
      detail.style.display = 'none';
      return;
    }

    row.classList.add('expanded');
    clearEl(detail);
    detail.style.display = 'block';
    detail.appendChild(h('div', { class: 'text-muted mono', style: { padding: '16px' } }, 'Loading unit data…'));
    this._loadUnitDetail(detail, change);
  },

  async _loadUnitDetail(detail, change) {
    const matchType = change.match_type || 'changed';
    const coreCaps = ['llvm.ir.summary', 'llvm.ir.function_stats', 'clang.diag.summary',
                      'llvm.obj.summary', 'llvm.remarks.summary', 'llvm.debug.summary'];

    const unitId = change.candidate_unit_id || change.base_unit_id;
    const snapId = change.candidate_unit_id ? this._candidateId : this._baseId;

    if (!unitId || !snapId) {
      clearEl(detail);
      detail.appendChild(h('div', { class: 'empty-state', style: { padding: '20px' } },
        h('div', {}, 'No unit data available')));
      return;
    }

    const res = await API.queryUnit(unitId, coreCaps);
    clearEl(detail);

    if (!res.ok || !Array.isArray(res.data) || !res.data.length) {
      detail.appendChild(h('div', { class: 'empty-state', style: { padding: '20px' } },
        h('div', {}, 'Could not load unit capabilities'),
        h('div', { class: 'reason' }, res.error || 'No data returned')));
      return;
    }

    const results = CapabilityData.normalizeResults(res.data[0]?.results || res.data);
    const available = results.filter(r => r.available);

    if (!available.length) {
      detail.appendChild(h('div', { class: 'empty-state', style: { padding: '20px' } },
        h('div', {}, 'No capability data available for this unit')));
      return;
    }

    const label = matchType === 'added' ? 'New Unit Capabilities' : matchType === 'removed' ? 'Removed Unit Capabilities' : 'Unit Capabilities';
    detail.appendChild(h('div', { style: { fontSize: '11px', textTransform: 'uppercase', letterSpacing: '.5px', color: 'var(--fg3)', padding: '0 16px 8px', borderBottom: '1px solid var(--border)', marginBottom: '12px' } }, label));

    available.forEach(result => {
      detail.appendChild(h('section', { class: 'capability-card', style: { marginBottom: '12px' } },
        h('div', { class: 'capability-card-title mono' }, friendlyCapabilityName(result.capability)),
        UI.capabilityPanel(result)
      ));
    });
  },
};

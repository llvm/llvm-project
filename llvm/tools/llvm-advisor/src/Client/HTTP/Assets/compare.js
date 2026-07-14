/* ============================================================
   LLVM Advisor — Compare View
   ============================================================ */

const CompareView = {
  _baseId: null,
  _candidateId: null,
  _results: null,
  _baseSummary: null,
  _candidateSummary: null,
  _remarksDiff: null,
  _remarkPage: 0,
  _remarkPageCount: 1,

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
    container.appendChild(h('div', { id: 'compare-remarks-diff' }));
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

    const [compareRes, baseSumRes, candSumRes, remDiffRes] = await Promise.all([
      API.compare(this._baseId, this._candidateId),
      API.snapshotSummary(this._baseId).catch(() => ({ ok: false, data: {} })),
      API.snapshotSummary(this._candidateId).catch(() => ({ ok: false, data: {} })),
      API.compareRemarks(this._baseId, this._candidateId, 0, 100).catch(() => ({ ok: false, data: null })),
    ]);

    this._baseSummary = baseSumRes.ok && baseSumRes.data ? baseSumRes.data : {};
    this._candidateSummary = candSumRes.ok && candSumRes.data ? candSumRes.data : {};
    this._remarksDiff = remDiffRes.ok && remDiffRes.data ? remDiffRes.data : null;
    this._remarkPage = 0;

    if (!compareRes.ok) {
      if (resultsEl) { clearEl(resultsEl); resultsEl.appendChild(UI.errorCard(compareRes.error || 'Compare failed', () => this.runCompare())); }
      return;
    }

    this._results = compareRes.data;
    this.renderMatchSummary(summaryEl);
    this.renderSummaryComparison(sumSectionEl);
    const remDiffEl = document.getElementById('compare-remarks-diff');
    if (remDiffEl) this.renderRemarksDiff(remDiffEl);
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

  renderRemarksDiff(el) {
    clearEl(el);
    const diff = this._remarksDiff;
    if (!diff) return;

    const s = diff.summary || {};
    const newMissed = s.new_missed || 0;
    const resolved = s.resolved_missed || 0;
    const changed = s.functions_changed || 0;
    const added = s.functions_added || 0;
    const removed = s.functions_removed || 0;

    if (changed === 0 && newMissed === 0 && resolved === 0) return;

    const section = h('div', { class: 'compare-remarks-section' });
    section.appendChild(h('h3', { style: { margin: '0 0 10px' } }, 'Optimization Impact'));

    const impactBar = h('div', { style: { display: 'flex', gap: '10px', marginBottom: '12px', flexWrap: 'wrap' } });
    if (newMissed > 0)
      impactBar.appendChild(h('div', { class: 'summary-metric warn' },
        h('div', { class: 'label' }, 'New Missed'),
        h('div', { class: 'values', style: { color: 'var(--orange)' } }, `+${formatNumber(newMissed)}`)
      ));
    if (resolved > 0)
      impactBar.appendChild(h('div', { class: 'summary-metric' },
        h('div', { class: 'label' }, 'Resolved'),
        h('div', { class: 'values', style: { color: 'var(--green)' } }, `-${formatNumber(resolved)}`)
      ));
    impactBar.appendChild(h('div', { class: 'summary-metric' },
      h('div', { class: 'label' }, 'Functions Changed'),
      h('div', { class: 'values' }, formatNumber(changed))
    ));
    if (added > 0)
      impactBar.appendChild(h('div', { class: 'summary-metric' },
        h('div', { class: 'label' }, 'Functions Added'),
        h('div', { class: 'values' }, formatNumber(added))
      ));
    if (removed > 0)
      impactBar.appendChild(h('div', { class: 'summary-metric' },
        h('div', { class: 'label' }, 'Functions Removed'),
        h('div', { class: 'values' }, formatNumber(removed))
      ));
    section.appendChild(impactBar);

    const functions = diff.functions || [];
    if (!functions.length) { el.appendChild(section); return; }

    const tbl = h('table', { class: 'top-units-table', style: { width: '100%' } });
    const thead = h('tr', {},
      h('th', {}, 'Function'),
      h('th', { style: { textAlign: 'right' } }, 'Before'),
      h('th', { style: { textAlign: 'right' } }, 'After'),
      h('th', { style: { textAlign: 'right' } }, '∆ Missed'),
      h('th', { style: { textAlign: 'right' } }, '∆ Total'),
    );
    tbl.appendChild(h('thead', {}, thead));
    const tbody = h('tbody', {});

    functions.forEach((fn, idx) => {
      const delta = fn.delta_missed;
      const color = delta > 0 ? 'var(--orange)' : delta < 0 ? 'var(--green)' : 'var(--fg)';
      const row = h('tr', { style: { cursor: 'pointer' },
        onClick: () => this._toggleFnDetail(fn, row, tbody, idx)
      },
        h('td', { class: 'mono', style: { fontSize: '11px', maxWidth: '300px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }, title: fn.name }, fn.name),
        h('td', { class: 'num' }, formatNumber(fn.before?.missed || 0)),
        h('td', { class: 'num' }, formatNumber(fn.after?.missed || 0)),
        h('td', { class: 'num', style: { color } }, `${delta >= 0 ? '+' : ''}${formatNumber(delta)}`),
        h('td', { class: 'num', style: { color: fn.delta_total > 0 ? 'var(--orange)' : fn.delta_total < 0 ? 'var(--green)' : '' } }, `${fn.delta_total >= 0 ? '+' : ''}${formatNumber(fn.delta_total)}`),
      );
      tbody.appendChild(row);
    });
    tbl.appendChild(tbody);

    const wrap = h('div', { class: 'top-units-wrap', style: { maxHeight: '300px', overflow: 'auto' } }, tbl);
    section.appendChild(wrap);

    const total = diff.total || 0;
    const pageSize = 100;
    this._remarkPageCount = Math.max(1, Math.ceil(total / pageSize));
    if (total > functions.length) {
      const pager = h('div', { style: { display: 'flex', gap: '8px', alignItems: 'center', marginTop: '8px', fontSize: '12px' } });
      const prevBtn = h('button', { class: 'triage-chip', onClick: async () => {
        if (this._remarkPage <= 0) return;
        this._remarkPage--;
        await this._fetchRemarkPage(el);
      }}, '← Prev');
      const nextBtn = h('button', { class: 'triage-chip', onClick: async () => {
        if (this._remarkPage >= this._remarkPageCount - 1) return;
        this._remarkPage++;
        await this._fetchRemarkPage(el);
      }}, 'Next →');
      const pageLabel = h('span', { class: 'text-muted' }, `Page ${this._remarkPage + 1} of ${this._remarkPageCount} (${formatNumber(total)} changed)`);
      pager.appendChild(prevBtn); pager.appendChild(pageLabel); pager.appendChild(nextBtn);
      section.appendChild(pager);
    }

    el.appendChild(section);
  },

  async _fetchRemarkPage(el) {
    const res = await API.compareRemarks(this._baseId, this._candidateId, this._remarkPage * 100, 100);
    if (res.ok && res.data) { this._remarksDiff = res.data; this.renderRemarksDiff(el); }
  },

  async _toggleFnDetail(fn, row, tbody, idx) {
    const existingId = `fn-detail-${idx}`;
    const existing = tbody.querySelector(`#${existingId}`);
    if (existing) { existing.remove(); row.classList.remove('expanded'); return; }
    row.classList.add('expanded');
    const detailRow = h('tr', { id: existingId });
    const cell = h('td', { colspan: '5', style: { padding: '8px 12px', background: 'var(--bg2)', fontSize: '11px' } });
    cell.textContent = 'Loading…';
    detailRow.appendChild(cell);
    row.after(detailRow);

    const res = await API.compareFunctionDetail(this._baseId, this._candidateId, fn.name);
    clearEl(cell);
    if (!res.ok) { cell.textContent = 'Failed to load detail.'; return; }
    const d = res.data;
    const added = d.added || [];
    const removed = d.removed || [];

    if (!added.length && !removed.length) { cell.textContent = 'No remark-level changes found.'; return; }

    const TYPE_NAMES = { 1: 'passed', 2: 'missed', 3: 'analysis', 6: 'failure' };
    const TYPE_COLORS = { 1: 'var(--green)', 2: 'var(--orange)', 3: 'var(--teal)', 6: 'var(--red)' };

    const makeEntries = (items, sign, color) => items.map(r => {
      const hasLoc = r.file && r.line > 0;
      const explorerLink = hasLoc
        ? h('a', {
            class: 'text-muted',
            style: { fontSize: '10px', cursor: 'pointer', textDecoration: 'underline' },
            onClick: (e) => {
              e.stopPropagation();
              State.set('currentSnapshot', State.get('snapshots').find(s => s.id === this._candidateId) || null);
              Router.navigate(`/explorer?path=${encodeURIComponent(r.file)}&line=${r.line}`);
            }
          }, `${r.file.split('/').pop()}:${r.line}`)
        : null;
      return h('div', { style: { display: 'flex', gap: '8px', padding: '2px 0', alignItems: 'baseline' } },
        h('span', { style: { color, fontWeight: '600', minWidth: '16px' } }, sign),
        h('span', { style: { color: TYPE_COLORS[r.type] || 'var(--fg3)', minWidth: '60px' } }, TYPE_NAMES[r.type] || '?'),
        h('span', { style: { fontWeight: '500' } }, r.name || ''),
        h('span', { class: 'text-muted' }, r.pass || ''),
        h('span', { style: { color: color, fontSize: '10px' } }, `×${Math.abs(r.delta || r.after_count - r.before_count)}`),
        explorerLink,
      );
    });

    if (removed.length) {
      cell.appendChild(h('div', { style: { marginBottom: '4px', fontWeight: '600', color: 'var(--red)' } }, 'Removed (resolved):'));
      makeEntries(removed, '−', 'var(--green)').forEach(e => cell.appendChild(e));
    }
    if (added.length) {
      cell.appendChild(h('div', { style: { marginTop: removed.length ? '8px' : 0, marginBottom: '4px', fontWeight: '600', color: 'var(--orange)' } }, 'Added (new):'));
      makeEntries(added, '+', 'var(--orange)').forEach(e => cell.appendChild(e));
    }
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

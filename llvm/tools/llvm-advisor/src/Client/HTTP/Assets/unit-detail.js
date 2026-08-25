/* ============================================================
   LLVM Advisor — Unit Detail View
   ============================================================ */

const UnitDetailView = {
  async render(params) {
    const unitId = params.id;
    const snapId = params.snapshot || State.get('currentSnapshot')?.id;
    if (!unitId || !snapId) {
      Shell.renderMain(h('div', { class: 'empty-state' }, h('div', {}, 'No unit selected')));
      return;
    }

    const container = h('div', { class: 'unit-detail-layout clean' });
    const sidebar = h('div', { class: 'unit-detail-sidebar' });
    const main = h('div', { class: 'unit-detail-main' });
    container.appendChild(sidebar);
    container.appendChild(main);

    Shell.renderMain(container);

    // Load unit data
    const unitRes = await API.unit(snapId, unitId);
    if (!unitRes.ok) {
      main.appendChild(h('div', { class: 'empty-state' },
        h('div', {}, 'Unit not found'),
        h('div', { class: 'reason' }, unitRes.error)
      ));
      return;
    }
    const unit = unitRes.data;

    const fileInfo = this.pathInfo(unit.source_path || unit.id);
    main.appendChild(h('div', { class: 'unit-detail-hero' },
      h('button', { class: 'back-link', onClick: () => Router.navigate('/units') }, 'Units'),
      h('div', { class: 'unit-title-block' },
        h('div', { class: 'unit-detail-title' }, fileInfo.file || unit.id),
        fileInfo.dir ? h('div', { class: 'unit-detail-path mono', title: unit.source_path || unit.id }, fileInfo.dir) : null
      ),
      h('div', { class: 'unit-detail-meta' },
        [unit.language, unit.target_triple, unit.toolchain_version, timeAgo(unit.created_unix)].filter(Boolean).map(v =>
          h('span', {}, v)
        )
      )
    ));

    const summary = h('div', { class: 'unit-detail-summary', id: 'unit-detail-summary' },
      this.summaryCard('Instructions', unit.instructions, 'neutral'),
      this.summaryCard('Symbols', unit.symbols, 'neutral'),
      this.summaryCard('Warnings', unit.warnings || 0, (unit.warnings || 0) ? 'warn' : 'neutral'),
      this.summaryCard('Errors', unit.errors || 0, (unit.errors || 0) ? 'danger' : 'neutral')
    );
    main.appendChild(summary);

    const tabState = { active: 'Overview', results: [], byCapability: new Map() };

    // Code viewer and tabs
    const tabs = ['Overview', 'Remarks', 'Artifacts'];
    const tabHeaders = h('div', { class: 'code-tabs' });
    const contentArea = h('div', { class: 'code-content', id: 'code-content' });
    const inlineExplorer = h('div', { id: 'inline-explorer' });

    const updateTab = (idx) => {
      tabState.active = tabs[idx];
      Array.from(tabHeaders.children).forEach((t, i) => t.classList.toggle('active', i === idx));
      clearEl(contentArea);
      contentArea.appendChild(this.renderTab(tabs[idx], tabState));
      // Close inline explorer when switching tabs
      clearEl(inlineExplorer);
    };

    tabs.forEach((name, i) => {
      const tab = h('div', { class: 'code-tab' + (i === 0 ? ' active' : ''), onClick: () => updateTab(i) }, name);
      tabHeaders.appendChild(tab);
    });

    const viewer = h('div', { class: 'code-viewer clean-viewer' }, tabHeaders, contentArea);
    main.appendChild(viewer);
    main.appendChild(inlineExplorer);

    this.renderCapSidebar(sidebar, unit);

    // Try to load capability data
    this.loadCapabilities(unit, sidebar, null, tabState, main, () => updateTab(tabs.indexOf(tabState.active)));
  },

  pathInfo(path) {
    const normalized = String(path || '').replace(/\\/g, '/');
    const idx = normalized.lastIndexOf('/');
    return idx >= 0
      ? { dir: normalized.slice(0, idx), file: normalized.slice(idx + 1) }
      : { dir: '', file: normalized };
  },

  summaryCard(label, value, tone) {
    return h('div', { class: `unit-detail-stat ${tone}` },
      h('span', { class: 'unit-detail-stat-value' }, formatNumber(value)),
      h('span', { class: 'unit-detail-stat-label' }, label)
    );
  },


  renderCapSidebar(sidebar, unit) {
    clearEl(sidebar);
    sidebar.appendChild(h('div', { class: 'unit-side-card' },
      h('div', { class: 'rail-title' }, 'Unit'),
      h('div', { class: 'soft-kv' }, h('span', {}, 'Language'), h('strong', {}, unit.language || '–')),
      h('div', { class: 'soft-kv' }, h('span', {}, 'Target'), h('strong', {}, unit.target_triple || '–')),
      h('div', { class: 'soft-kv' }, h('span', {}, 'Toolchain'), h('strong', {}, unit.toolchain_version || '–'))
    ));
    sidebar.appendChild(h('div', { class: 'unit-side-card', id: 'capability-health-card' },
      h('div', { class: 'rail-title' }, 'Coverage'),
      h('div', { class: 'rail-empty' }, 'Loading analysis coverage')
    ));

  renderTab(tab, state) {
    if (!state.results.length)
      return h('div', { class: 'empty-state', style: { padding: '40px' } },
        h('div', {}, 'Loading capabilities'),
        h('div', { class: 'reason mono' }, 'Querying analyzer results for this unit'));

    if (tab === 'Remarks') {
      const relResult = state.byCapability.get('llvm.remarks.relational');
      if (relResult && relResult.available && relResult.value && relResult.value.columns) {
        const v = relResult.value;
        const rel = { count: v.count || 0, columns: v.columns || {}, strings: v.strings || {} };
        const total = rel.count || 0;
        const filtered = new Int32Array(total);
        for (let i = 0; i < total; i++) filtered[i] = i;
        return RemarksView._renderTriageGrid(rel, null, filtered, total);
      }
      const findings = state.results
        .filter(r => r.capability.includes('remarks'))
        .flatMap(r => r.findings);
      if (!findings.length) {
        return this.emptyTab('No optimization remarks', 'No optimization remarks were reported for this unit.');
      }
      return h('div', { class: 'capability-stack' },
        UI.passTimeline(findings),
        UI.findingList(findings)
      );
    }

    if (tab === 'Artifacts') {
      const artifacts = state.results.flatMap(r => r.artifacts.map(a => ({ capability: r.capability, ...a })));
      if (!artifacts.length)
        return this.emptyTab('No artifacts', 'Capabilities returned metrics, tables, or findings only.');
      return h('div', { class: 'artifact-stack' }, artifacts.map(a => {
        const lang = a.name === 'ir' || a.name === 'llvm' ? 'ir' : (a.name === 'assembly' || a.name === 'asm') ? 'asm' : null;
        return h('div', { class: 'artifact-block' },
          h('div', { class: 'table-title' }, `${friendlyCapabilityName(a.capability)} · ${a.name}`),
          a.text && lang ? Syntax.codeBlock(a.text, lang) : a.text ? h('pre', { class: 'raw-json' }, a.text) : h('pre', { class: 'raw-json' }, JSON.stringify(a.data, null, 2))
        );
      }));
    }

    const metrics = this.collectOverview(state.results);
    return h('div', { class: 'unit-overview-panel' },
      h('div', { class: 'quiet-section-title' }, 'Summary'),
      h('div', { class: 'unit-overview-cards' },
        this.summaryCard('Remarks', metrics.remarks, metrics.remarks ? 'info' : 'neutral')
      ),
      h('div', { class: 'quiet-section-title' }, 'Available Analysis'),
      h('div', { class: 'analysis-list' }, state.results.map(r =>
        h('button', {
          class: `analysis-row ${r.available ? 'available' : 'missing'}`,
          title: r.reason || r.capability,
          onClick: () => Shell.renderDetail(UI.capabilityPanel(r), friendlyCapabilityName(r.capability)),
        },
          h('span', { class: 'analysis-name' }, friendlyCapabilityName(r.capability)),
          h('span', { class: 'analysis-state' }, r.available ? 'Ready' : 'Missing')
        )
      ))
    );
  },

  emptyTab(title, reason) {
    return h('div', { class: 'empty-state soft-empty' },
      h('div', {}, title),
      h('div', { class: 'reason' }, reason));
  },

  collectOverview(results) {
    const metrics = { remarks: 0 };
    results.forEach(r => {
      if (!r.available) return;
      metrics.remarks += Number(r.metrics.count && r.capability.includes('remarks') ? r.metrics.count : 0);
    });
    return metrics;
  },

  async loadCapabilities(unit, sidebar, pills, tabState, main, refresh) {
    const capRes = await API.capabilities();
    const caps = Array.isArray(capRes.data)
      ? capRes.data.filter(spec => CapabilityData.shouldQueryCapability(spec, 'unit')).map(c => c.id).filter(Boolean)
      : ['llvm.remarks.summary', 'llvm.remarks.detail', 'llvm.remarks.relational', 'llvm.remarks.hotspot'];
    const res = await API.queryUnit(unit.id, caps);
    if (!res.ok) {
      if (main) main.appendChild(UI.errorCard(res.error || 'query failed', () => this.render({ id: unit.id, snapshot: unit.snapshot_id || State.get('currentSnapshot')?.id })));
      return;
    }

    const results = CapabilityData.normalizeResults(res.data);
    tabState.results = results;
    tabState.byCapability = new Map(results.map(r => [r.capability, r]));
    this.renderCoverage(sidebar, results);

    if (refresh) refresh();
  },

  renderCoverage(sidebar, results) {
    const card = sidebar.querySelector('#capability-health-card');
    if (!card) return;
    clearEl(card);
    const available = results.filter(r => r.available).length;
    card.appendChild(h('div', { class: 'rail-title' }, 'Coverage'));
    card.appendChild(h('div', { class: 'coverage-number' },
      h('strong', {}, `${available}/${results.length}`),
      h('span', {}, ' analyses ready')
    ));
    const list = h('div', { class: 'coverage-list' });
    results.slice(0, 10).forEach(r => list.appendChild(h('div', { class: `coverage-row ${r.available ? 'available' : 'missing'}` },
      h('span', {}, friendlyCapabilityName(r.capability)),
      h('i', {})
    )));
    card.appendChild(list);
  },
};
